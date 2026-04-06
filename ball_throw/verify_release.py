"""
verify_release.py
=================
Verifies whether the robot actually achieved the desired
P_RELEASE and V_EE_RELEASE — both offline and online.

OFFLINE check (before execution):
    Uses Pinocchio FK + Jacobian on the IK solution.
    Answers: "How accurately did IK solve the problem?"

ONLINE check (after execution):
    Reads actual_trajectory.csv (logged by the C++ executor).
    Extracts the row at release_index.
    Applies FK + Jacobian to the measured joint state.
    Answers: "What did the robot actually achieve at release time?"

Usage:
    # Offline only (just checks IK accuracy)
    python verify_release.py --offline

    # Online (checks actual robot state at release)
    python verify_release.py --online

    # Both
    python verify_release.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

# ─────────────────────────────────────────────────────────────────────────────
#  Pinocchio helpers (duplicated here so this file is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _pad(model, q7):
    extra = getattr(model, "arm_extra_dof", 0)
    return np.concatenate([q7, np.zeros(extra)]) if extra > 0 else q7


def _load_model(urdf_path: str):
    if not HAS_PINOCCHIO:
        return None, None
    urdf = Path(urdf_path)
    if not urdf.exists():
        log.error("URDF not found: %s", urdf)
        return None, None
    model, _, _ = pin.buildModelsFromUrdf(str(urdf), str(urdf.parent))
    model.arm_extra_dof = model.nq - 7
    data = model.createData()
    return model, data


def _ee_position(model, data, q7: np.ndarray) -> np.ndarray:
    """Forward kinematics → end-effector position [3]."""
    q_full = _pad(model, q7)
    pin.forwardKinematics(model, data, q_full)
    pin.framesForwardKinematics(model, data, q_full)
    for name in ("panda_hand", "panda_link8"):
        fid = model.getFrameId(name)
        if fid < model.nframes:
            return np.array(data.oMf[fid].translation)
    raise RuntimeError("EE frame not found")


def _ee_velocity(model, data, q7: np.ndarray,
                 dq7: np.ndarray) -> np.ndarray:
    """Jacobian · dq → end-effector linear velocity [3]."""
    q_full  = _pad(model, q7)
    pin.computeJointJacobians(model, data, q_full)
    pin.framesForwardKinematics(model, data, q_full)
    for name in ("panda_hand", "panda_link8"):
        fid = model.getFrameId(name)
        if fid < model.nframes:
            J = pin.getFrameJacobian(
                model, data, fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            return (J[:3, :7] @ dq7)
    raise RuntimeError("EE frame not found")


# ─────────────────────────────────────────────────────────────────────────────
#  Offline verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_offline(meta_path: str, urdf_path: str) -> dict:
    """
    Check how accurately IK solved the problem.

    Loads q_release, dq_release from metadata.
    Applies FK and Jacobian.
    Returns error metrics.
    """
    meta = np.load(meta_path, allow_pickle=True).item()
    q_rel  = np.array(meta["Q_RELEASE"])
    dq_rel = np.array(meta["DQ_RELEASE"])

    # These are the targets the user specified
    p_target = np.array(meta.get("P_RELEASE",  [np.nan]*3))
    v_target = np.array(meta.get("V_EE_RELEASE", [np.nan]*3))

    result = {
        "q_release":  q_rel,
        "dq_release": dq_rel,
        "p_target":   p_target,
        "v_target":   v_target,
    }

    if not HAS_PINOCCHIO:
        log.warning("Pinocchio not available — cannot compute FK offline")
        result["pinocchio_available"] = False
        return result

    model, data = _load_model(urdf_path)
    if model is None:
        result["pinocchio_available"] = False
        return result

    result["pinocchio_available"] = True
    result["p_achieved_ik"] = _ee_position(model, data, q_rel)
    result["v_achieved_ik"] = _ee_velocity(model, data, q_rel, dq_rel)

    # Check whether task-space targets were recorded (NaN in mock mode)
    has_target = not (np.any(np.isnan(p_target)) or np.any(np.isnan(v_target)))
    result["has_target"] = has_target

    if has_target:
        result["p_error_ik"]   = result["p_achieved_ik"] - p_target
        result["v_error_ik"]   = result["v_achieved_ik"] - v_target
        result["p_error_norm"] = float(np.linalg.norm(result["p_error_ik"]))
        result["v_error_norm"] = float(np.linalg.norm(result["v_error_ik"]))
    else:
        result["p_error_ik"]   = np.full(3, np.nan)
        result["v_error_ik"]   = np.full(3, np.nan)
        result["p_error_norm"] = np.nan
        result["v_error_norm"] = np.nan

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Online verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_online(meta_path: str,
                  actual_csv_path: str,
                  urdf_path: str) -> dict:
    """
    Check what the robot actually achieved at the release instant.

    Reads actual_trajectory.csv (written by the C++ executor).
    Extracts the row at release_index.
    Applies FK and Jacobian to measured q, dq.
    Returns error metrics.

    actual_trajectory.csv columns (36):
        t | q0..q6 | dq0..dq6 | tau_ext0..tau_ext6
        | tau_cmd0..tau_cmd6 | sigma_min | tau_ext_max
    """
    meta = np.load(meta_path, allow_pickle=True).item()
    ri       = int(meta["release_index"])
    p_target = np.array(meta.get("P_RELEASE",    [np.nan]*3))
    v_target = np.array(meta.get("V_EE_RELEASE", [np.nan]*3))
    q_planned  = np.array(meta["Q_RELEASE"])
    dq_planned = np.array(meta["DQ_RELEASE"])

    actual_csv = Path(actual_csv_path)
    if not actual_csv.exists():
        return {
            "error": (
                f"actual_trajectory.csv not found: {actual_csv}\n"
                "Run the C++ executor first — it writes this file automatically."
            )
        }

    actual = np.loadtxt(str(actual_csv), delimiter=",", skiprows=1)

    # The actual CSV only covers the tracking phase (not the ramp).
    # Row i in actual CSV corresponds to trajectory sample i.
    # Release is at ri in the planned trajectory = ri in the actual CSV.
    if ri >= len(actual):
        return {
            "error": (
                f"release_index={ri} is beyond actual CSV length={len(actual)}. "
                "Was the trajectory completed fully?"
            )
        }

    row     = actual[ri]
    t_rel   = row[0]
    q_meas  = row[1:8]
    dq_meas = row[8:15]

    result = {
        "t_release":   t_rel,
        "q_planned":   q_planned,
        "dq_planned":  dq_planned,
        "q_measured":  q_meas,
        "dq_measured": dq_meas,
        "q_tracking_error":  q_meas  - q_planned,
        "dq_tracking_error": dq_meas - dq_planned,
        "p_target": p_target,
        "v_target": v_target,
    }

    if not HAS_PINOCCHIO:
        log.warning("Pinocchio not available — cannot compute FK online")
        result["pinocchio_available"] = False
        return result

    model, data = _load_model(urdf_path)
    if model is None:
        result["pinocchio_available"] = False
        return result

    result["pinocchio_available"] = True

    # FK on planned release state
    result["p_planned_fk"] = _ee_position(model, data, q_planned)
    result["v_planned_fk"] = _ee_velocity(model, data, q_planned, dq_planned)

    # FK on measured release state
    result["p_actual"]     = _ee_position(model, data, q_meas)
    result["v_actual"]     = _ee_velocity(model, data, q_meas,  dq_meas)

    # Errors
    result["p_error_vs_target"]  = result["p_actual"] - p_target
    result["v_error_vs_target"]  = result["v_actual"] - v_target
    result["p_error_norm"]       = float(np.linalg.norm(result["p_error_vs_target"]))
    result["v_error_norm"]       = float(np.linalg.norm(result["v_error_vs_target"]))

    result["p_tracking_error"]   = result["p_actual"]    - result["p_planned_fk"]
    result["v_tracking_error"]   = result["v_actual"]    - result["v_planned_fk"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _vec(v, decimals=4):
    return "[" + "  ".join(f"{x:+.{decimals}f}" for x in v) + "]"


def print_offline_report(r: dict) -> None:
    print()
    print("=" * 60)
    print("  OFFLINE VERIFICATION  (IK accuracy)")
    print("=" * 60)

    if not r.get("pinocchio_available", True):
        print("  [SKIP] Pinocchio not available — install with: pip install pin")
        return

    if "error" in r:
        print(f"  [ERROR] {r['error']}")
        return

    has_target = r.get("has_target", False)

    print(f"\n  End-effector POSITION at release:")
    print(f"    FK result: {_vec(r['p_achieved_ik'])} m")
    if has_target:
        p_ok = r["p_error_norm"] < 1e-3
        print(f"    Target   : {_vec(r['p_target'])} m")
        print(f"    Error    : {_vec(r['p_error_ik'])} m")
        print(f"    ||error||: {r['p_error_norm']*1000:.3f} mm  "
              f"{'✓ OK' if p_ok else '✗ IK did not converge well'}")
    else:
        p_ok = True
        print(f"    Target   : N/A  (mock mode — no task-space target was set)")

    print(f"\n  End-effector VELOCITY at release:")
    print(f"    FK result: {_vec(r['v_achieved_ik'])} m/s  (= J · dq_release)")
    if has_target:
        v_ok = r["v_error_norm"] < 1e-2
        print(f"    Target   : {_vec(r['v_target'])} m/s")
        print(f"    Error    : {_vec(r['v_error_ik'])} m/s")
        print(f"    ||error||: {r['v_error_norm']*1000:.3f} mm/s  "
              f"{'✓ OK' if v_ok else '✗ Velocity IK inaccurate'}")
    else:
        v_ok = True
        print(f"    Target   : N/A  (mock mode — no task-space target was set)")

    print()
    if not has_target:
        print("  NOTE: Running in mock mode — joint-space values were set directly,")
        print("        not derived from IK. The FK result above shows where the")
        print("        end-effector will actually be at the release instant.")
        print("        To verify IK accuracy, run:  python main_throw.py  (full mode)")
    elif p_ok and v_ok:
        print("  RESULT: IK solved accurately. The planned trajectory will")
        print("          reach P_RELEASE and V_EE_RELEASE if tracking is good.")
    else:
        print("  RESULT: IK accuracy is insufficient.")
        print("          Increase IK iterations or improve Q_RELEASE_INIT guess.")
    print("=" * 60)


def print_online_report(r: dict) -> None:
    print()
    print("=" * 60)
    print("  ONLINE VERIFICATION  (actual robot at release instant)")
    print("=" * 60)

    if not r.get("pinocchio_available", True):
        print("  [SKIP] Pinocchio not available — install with: pip install pin")
        return

    if "error" in r:
        print(f"\n  [ERROR] {r['error']}")
        return

    p_ok = r["p_error_norm"] < 2e-3    # 2 mm
    v_ok = r["v_error_norm"] < 0.05    # 5 cm/s

    print(f"\n  Release instant: t = {r['t_release']:.4f} s")

    print(f"\n  Joint position tracking at release:")
    print(f"    Planned  : {_vec(r['q_planned'])}")
    print(f"    Measured : {_vec(r['q_measured'])}")
    print(f"    Error    : {_vec(r['q_tracking_error'])} rad")
    print(f"    Max |err|: {np.max(np.abs(r['q_tracking_error']))*1000:.2f} mrad")

    print(f"\n  Joint velocity tracking at release:")
    print(f"    Planned  : {_vec(r['dq_planned'])}")
    print(f"    Measured : {_vec(r['dq_measured'])}")
    print(f"    Error    : {_vec(r['dq_tracking_error'])} rad/s")
    print(f"    Max |err|: {np.max(np.abs(r['dq_tracking_error']))*1000:.2f} mrad/s")

    print(f"\n  End-effector POSITION at release:")
    print(f"    Target         : {_vec(r['p_target'])} m")
    print(f"    Planned (FK)   : {_vec(r['p_planned_fk'])} m")
    print(f"    Actual (FK)    : {_vec(r['p_actual'])} m")
    print(f"    Error vs target: {_vec(r['p_error_vs_target'])} m")
    print(f"    ||error||      : {r['p_error_norm']*1000:.3f} mm  "
          f"{'✓ OK' if p_ok else '✗ Position error too large'}")

    print(f"\n  End-effector VELOCITY at release:")
    print(f"    Target         : {_vec(r['v_target'])} m/s")
    print(f"    Planned (J·dq) : {_vec(r['v_planned_fk'])} m/s")
    print(f"    Actual (J·dq)  : {_vec(r['v_actual'])} m/s")
    print(f"    Error vs target: {_vec(r['v_error_vs_target'])} m/s")
    print(f"    ||error||      : {r['v_error_norm']*1000:.3f} mm/s  "
          f"{'✓ OK' if v_ok else '✗ Velocity error too large'}")

    print()
    if p_ok and v_ok:
        print("  RESULT: Robot achieved the desired release state.")
        print("          Ball trajectory should match prediction.")
    else:
        reasons = []
        if not p_ok:
            reasons.append(
                f"position error {r['p_error_norm']*1000:.1f} mm > 2 mm threshold"
            )
        if not v_ok:
            reasons.append(
                f"velocity error {r['v_error_norm']*1000:.1f} mm/s > 50 mm/s threshold"
            )
        print("  RESULT: Release state NOT achieved accurately.")
        for r_ in reasons:
            print(f"    - {r_}")
        print()
        print("  Possible fixes:")
        print("    - Increase PD gains (Kp, Kd) in panda_executor.cpp")
        print("    - Use the official franka_description URDF for better")
        print("      RNEA accuracy (reduces PD correction needed)")
        print("    - Slow the trajectory down (larger T) to reduce")
        print("      the inertial terms that PyBullet gets wrong")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_verification(r_offline: dict | None,
                      r_online:  dict | None,
                      save_path: str | None = None) -> None:
    """
    Bar chart comparing target, planned, and actual EE position and velocity.
    """
    import matplotlib.pyplot as plt

    has_offline = (r_offline is not None and
                   r_offline.get("pinocchio_available") and
                   "p_achieved_ik" in r_offline)
    has_online  = (r_online  is not None and
                   r_online.get("pinocchio_available") and
                   "p_actual" in r_online)

    if not has_offline and not has_online:
        print("[PLOT] Nothing to plot — Pinocchio not available")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Release state verification — target vs planned (IK) vs actual (robot)",
        fontsize=13, fontweight="500",
    )

    axes_labels = ["X", "Y", "Z"]
    x = np.arange(3)
    w = 0.25

    for row, (qty, unit, target_key, ik_key, act_key) in enumerate([
        ("Position",  "m",   "p_target", "p_achieved_ik", "p_actual"),
        ("Velocity",  "m/s", "v_target", "v_achieved_ik", "v_actual"),
    ]):
        for col, component in enumerate(range(3)):
            ax = axes[row][col]

            groups = []
            labels = []
            colors = []

            target = (r_offline or r_online).get(target_key, [0,0,0])
            groups.append(target[component]);  labels.append("Target");  colors.append("#6B7280")

            if has_offline:
                val = r_offline[ik_key][component]
                groups.append(val); labels.append("IK planned"); colors.append("#2563EB")

            if has_online:
                val = r_online[act_key][component]
                groups.append(val); labels.append("Robot actual"); colors.append("#DC2626")

            bars = ax.bar(np.arange(len(groups)), groups,
                          color=colors, alpha=0.75, width=0.55,
                          edgecolor="white")
            for bar, val in zip(bars, groups):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + abs(bar.get_height())*0.02 + 1e-4,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(np.arange(len(groups)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(f"{qty} — {axes_labels[component]} component [{unit}]",
                         fontsize=9)
            ax.set_ylabel(unit, fontsize=8)
            ax.grid(True, axis="y", lw=0.3, alpha=0.5)
            ax.spines[["top","right"]].set_visible(False)

            # Error annotations
            if has_offline:
                err = r_offline[ik_key][component] - target[component]
                ax.annotate(f"IK err:\n{err*1000:.2f} mm" if row==0
                            else f"IK err:\n{err*1000:.1f} mm/s",
                            xy=(1, r_offline[ik_key][component]),
                            xytext=(1.3, r_offline[ik_key][component]),
                            fontsize=6.5, color="#2563EB")

            if has_online:
                err = r_online[act_key][component] - target[component]
                col_idx = 2 if has_offline else 1
                ax.annotate(f"Actual err:\n{err*1000:.2f} mm" if row==0
                            else f"Actual err:\n{err*1000:.1f} mm/s",
                            xy=(col_idx, r_online[act_key][component]),
                            xytext=(col_idx+0.2, r_online[act_key][component]),
                            fontsize=6.5, color="#DC2626")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Verification plot saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Verify P_RELEASE and V_EE_RELEASE were achieved"
    )
    parser.add_argument("--meta",   default="throw_trajectory_meta.npy",
                        help="Metadata file from main_throw.py")
    parser.add_argument("--actual", default="actual_trajectory.csv",
                        help="Actual trajectory CSV from C++ executor")
    parser.add_argument("--urdf",   default=None,
                        help="Path to panda.urdf (auto-detected from pybullet if omitted)")
    parser.add_argument("--offline", action="store_true",
                        help="Run offline IK accuracy check only")
    parser.add_argument("--online",  action="store_true",
                        help="Run online robot tracking check only")
    parser.add_argument("--plot",   action="store_true",
                        help="Show verification bar chart")
    parser.add_argument("--save",   default=None,
                        help="Save verification plot to this path")
    args = parser.parse_args()

    # Default: run both
    run_offline = args.offline or (not args.offline and not args.online)
    run_online  = args.online  or (not args.offline and not args.online)

    # Auto-detect URDF
    urdf = args.urdf
    if urdf is None:
        try:
            import pybullet_data, os
            urdf = os.path.join(pybullet_data.getDataPath(),
                                "franka_panda/panda.urdf")
        except ImportError:
            urdf = "franka_panda/panda.urdf"

    r_offline, r_online = None, None

    if run_offline:
        r_offline = verify_offline(args.meta, urdf)
        print_offline_report(r_offline)

    if run_online:
        r_online = verify_online(args.meta, args.actual, urdf)
        print_online_report(r_online)

    if args.plot or args.save:
        plot_verification(r_offline, r_online, save_path=args.save)


if __name__ == "__main__":
    main()